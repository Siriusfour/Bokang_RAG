package global

import "errors"

// Auth 错误
var (
	//Token
	ErrTokenInvalid      = errors.New("token无效")
	ErrTokenExpired      = errors.New("token已过期")
	ErrTokenMalformed    = errors.New("token格式错误")
	ErrTokenNotActiveYet = errors.New("token尚未生效")
	ErrSessionExpired    = errors.New("会话已失效")
	ErrUnauthorized      = errors.New("未授权")
	ErrAccountKicked     = errors.New("异地登陆")

	//password
	ErrFailLoginInfo = errors.New("请先输入登录信息")

	//OAuth
	ErrorOAuthIDrNotFound = errors.New("没有该条OAuthID记录")

	// webAthun错误
	ErrorClientDataTypeIsFail      = errors.New("type不为webauthn.create！")
	ErrorClientDataChallengeIsFail = errors.New("challenge错误！")
	ErrorClientDataRPID_IsFail     = errors.New("域名错误，当前页面域名与配置文件的RPID不符！")
)

// 基础错误
var (
	ErrorAssert        = errors.New("类型断言失败")
	ErrorDB_FindIsFail = errors.New("数据库查找失败")
)

// 雪花算法
var ()

const (
	ServerOccurredErrorMsg string = "服务器内部发生代码执行错误, "
	//系统部分
	ErrorsContainerKeyAlreadyExists string = "该键已经注册在容器中了"
	ErrorsPublicNotExists           string = "public 目录不存在"
	ErrorsConfigYamlNotExists       string = "配置文件不存在:"
	ErrorsConfigGormNotExists       string = "gorm_v2.yml 配置文件不存在"
	ErrorsStorageLogsNotExists      string = "storage/logs 目录不存在"
	ErrorsConfigInitFail            string = "初始化配置文件发生错误"
	ErrorsSoftLinkCreateFail        string = "自动创建软连接失败,请以管理员身份运行客户端(开发环境为goland等，生产环境检查命令执行者权限), " +
		"最后一个可能：如果您是360用户，请退出360相关软件，才能保证go语言创建软连接函数： os.Symlink() 正常运行"
	ErrorsSoftLinkDeleteFail string = "删除软软连接失败"
	ErrorGinSetTrustProxy    string = "Gin 设置信任代理服务器出错"

	ErrorsFuncEventAlreadyExists   string = "注册函数类事件失败，键名已经被注册"
	ErrorsFuncEventNotRegister     string = "没有找到键名对应的函数"
	ErrorsFuncEventNotCall         string = "注册的函数无法正确执行"
	ErrorsBasePath                 string = "初始化项目根目录失败"
	ErrorsTokenBaseInfo            string = "token最基本的格式错误,请提供一个有效的token!"
	ErrorsNoAuthorization          string = "token鉴权未通过，请通过token授权接口重新获取token,"
	ErrorsRefreshTokenFail         string = "token不符合刷新条件,请通过登陆接口重新获取token!"
	ErrorsParseTokenFail           string = "解析token失败"
	ErrorsGormInitFail             string = "Gorm 数据库驱动、连接初始化失败"
	ErrorsCasbinNoAuthorization    string = "Casbin 鉴权未通过，请在后台检查 casbin 设置参数"
	ErrorsGormNotInitGlobalPointer string = "%s 数据库全局变量指针没有初始化，请在配置文件 Config/gorm_v2.yml 设置 Gormv2.%s.IsInitGlobalGormMysql = 1, 并且保证数据库配置正确 \n"
	// 数据库部分
	ErrorsDbDriverNotExists        string = "数据库驱动类型不存在,目前支持的数据库类型：mysql、sqlserver、postgresql，您提交数据库类型："
	ErrorsDialectorDbInitFail      string = "gorm dialector 初始化失败,dbType:"
	ErrorsGormDBCreateParamsNotPtr string = "gorm Create 函数的参数必须是一个指针"
	ErrorsGormDBUpdateParamsNotPtr string = "gorm 的 Update、Save 函数的参数必须是一个指针(GinSkeleton ≥ v1.5.29 版本新增验证，为了完美支持 gorm 的所有回调函数,请在参数前面添加 & )"

	//redis部分
	ErrorsRedisInitConnFail string = "初始化redis连接池失败"
	ErrorsRedisAuthFail     string = "Redis AuthDTO 鉴权失败，密码错误"
	ErrorsRedisGetConnFail  string = " 从连接池获取一个连接失败，超过最大重试次数"
	ErrorRedisGetDATA       string = "Redis获取数据失败:"
	ErrorRedisGetKeyExist   string = "调用global.RedisClient.Exists失败"

	// 表单参数验证器未通过时的错误
	ErrorsValidatorNotExists      string = "不存在的验证器"
	ErrorsValidatorTransInitFail  string = "validator的翻译器初始化错误"
	ErrorNotAllParamsIsBlank      string = "该接口不允许所有参数都为空,请按照接口要求提交必填参数"
	ErrorsValidatorBindParamsFail string = "验证器绑定参数失败"
	ValidatorParamsCheckFailMsg   string = "参数校验失败"

	//token部分

	//snowflake
	ErrorsSnowflakeGetIdFail string = "获取snowflake唯一ID过程发生错误"

	// websocket
	ErrorsWebsocketOnOpenFail                 string = "websocket onopen 发生阶段错误"
	ErrorsWebsocketUpgradeFail                string = "websocket Upgrade 协议升级, 发生错误"
	ErrorsWebsocketReadMessageFail            string = "websocket ReadPump(实时读取消息)协程出错"
	ErrorsWebsocketBeatHeartFail              string = "websocket BeatHeart心跳协程出错"
	ErrorsWebsocketBeatHeartsMoreThanMaxTimes string = "websocket BeatHeart 失败次数超过最大值"
	ErrorsWebsocketClientOfflineTimeout       string = "websocket 客户端响应ping消息超过服务端允许的最长时间(秒):"
	ErrorsWebsocketSetWriteDeadlineFail       string = "websocket  设置消息写入截止时间出错"
	ErrorsWebsocketWriteMgsFail               string = "websocket  Write Msg(send msg) 失败"
	ErrorsWebsocketStateInvalid               string = "websocket  state 状态已经不可用(掉线、卡死等愿意，造成双方无法进行数据交互)"
	// rabbitMq
	ErrorsRabbitMqReconnectFail string = "RabbitMq消费者端掉线后重连失败，超过尝试最大次数"

	//文件上传
	ErrorsFilesUploadOpenFail string = "打开文件失败，详情："
	ErrorsFilesUploadReadFail string = "读取文件32字节失败，详情："

	// casbin 初始化可能的错误
	ErrorCasbinCanNotUseDbPtr         string = "casbin 的初始化基于gorm 初始化后的数据库连接指针，程序检测到 gorm 连接指针无效，请检查数据库配置！"
	ErrorCasbinCreateAdaptFail        string = "casbin NewAdapterByDBUseTableName 发生错误："
	ErrorCasbinCreateEnforcerFail     string = "casbin NewEnforcer 发生错误："
	ErrorCasbinNewModelFromStringFail string = "NewModelFromString 调用时出错："

	//Password部分
	ErrorPasswordOrSaltIsEmpty string = "密码不可为空！"
	ErrorPasswordHashIsFail    string = "hash处理失败："

	//业务部分
	ErrorRegisterIsFail     string = "注册失败:"
	ErrorGetUserIDIsFail    string = "获取用户ID失败："
	ErrorGetChallengeIsFail string = "获取随机挑战失败"
)
